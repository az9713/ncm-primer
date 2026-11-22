# Config reference

This page lists every field in `config/eval/*.yaml`. The "Read at" column gives the line that reads the field. "Not read" means that no Python file in the repository reads the field (searched with grep on 2026-11-21).

This reference tracks `config/eval/*.yaml`, `scripts/eval_shard.py`, `scripts/eval_merge.py`, `evaluate.py`, `models/*.py` and `tokenization/*.py`. Update them together.

The 9 config files for released models were compared with the `config.yaml` files on Hugging Face on 2026-11-21. All 9 were identical. The `owt` checkpoint folder had no `config.yaml`.

## Top level

| Field | Type | Configs | Read at | Meaning |
|-------|------|---------|---------|---------|
| `task` | string | all | `evaluate.py:23`, `scripts/eval_merge.py:21` | Name for the default output folder `eval/<task>` and for printed messages. |

## `checkpoints`

| Field | Type | Configs | Read at | Meaning |
|-------|------|---------|---------|---------|
| `generator` | path | all | `scripts/eval_shard.py:52` | Generator file, relative to `--checkpoint-dir`. |
| `tokenizer` | path | all | `scripts/eval_shard.py:45` | Tokenizer file, relative to `--checkpoint-dir`. LM1B uses `../tokenizer.pt`. |
| `bpe_vocab` | path | `tinystories` | `scripts/eval_shard.py:46-48` | BPE vocabulary pickle. Overwrites `pre_tokenizer.vocab_file_path`. |
| `bpe_merges` | path | `tinystories` | `scripts/eval_shard.py:46-48` | BPE merges pickle. Overwrites `pre_tokenizer.merges_file_path`. |

## `data`

| Field | Type | Configs | Read at | Meaning |
|-------|------|---------|---------|---------|
| `data` (empty `{}`) | map | `lm1b-*`, `owt`, `tinystories` | `scripts/eval_shard.py:79` | No entry for the split, so the task is unconditional. |
| `test` | path | `xsum`, `wmt14`, `sudoku-*` | `scripts/eval_shard.py:87` | Input pickle, relative to `--data-dir`. |
| `test_refs` | path | `xsum`, `wmt14`, `sudoku-*` | `scripts/eval_shard.py:88` | Reference pickle. The key is `<split>_refs`. |
| `validation` | path | `xsum` | Not read with the default split | With `--split validation`, the script also needs `validation_refs`, which no config has. It then raises a key error. |

## `model`

| Field | Type | Configs | Read at | Meaning |
|-------|------|---------|---------|---------|
| `n_layer` | int | all | `models/next_concept.py:73`, `:204` | Number of transformer blocks. |
| `n_head` | int | all | `models/common.py:59-69` | Attention heads. |
| `n_embd` | int | all | `models/next_concept.py:72` | Width of the generator. |
| `bias` | bool | all | `models/next_concept.py:75`, `models/common.py:65-66` | Bias in linear layers and norms. |
| `dropout` | float | all | `models/next_concept.py:74`, `models/common.py:67-71` | Dropout. Inactive in eval mode. |
| `context_length` | int | all | Not read | |
| `prefix_len` | int | all | `models/next_concept.py:92`, `scripts/eval_shard.py:55` | P, the prefix length. 1 for unconditional configs, 512 XSum, 96 WMT14, 91 Sudoku. |
| `raw_token_prefix` | bool | all | `models/next_concept.py:549` | `true`: embed the prefix with the tokenizer encoder `wte` only (`:555`). |
| `prefix_pad_token_id` | int | all except `sudoku-*` | `models/next_concept.py:246-247`, `:573-579`, `:531` | Pad id. Pad positions get −inf attention. Needed for guidance. |
| `vocab_size` | list of int | all | `models/next_concept.py:76`, `:208-212` | Codebook size per predicted level. Length K−1, or K with `query_level0`. |
| `predict_level0` | bool | 7 configs | `models/next_concept.py:249` | `true` (WMT14): a head on the prefix predicts level 1. |
| `query_level0` | bool | 9 configs | `models/next_concept.py:88-91` | `true`: the model has learned level-1 queries and a head for level 1. |
| `attn_pattern` | enum | all | `models/next_concept.py:77`, `:266-276` | `block_diagonal` (all configs) or `block_causal`. |
| `use_positional_encoding` | bool | all | Not read | Positions come from `use_rope`. |
| `use_level_encoding` | bool | all | `models/next_concept.py:197`, `:562` | Add a learned vector per level. |
| `use_rope` | bool | all | `models/next_concept.py:95` | Rotary positions, restarted per level. |
| `rope_base` | int | all | `models/next_concept.py:96` | RoPE base: 10000 (LM1B, Sudoku) or 100000. |
| `use_qk_norm` | bool | all | `models/common.py:77-81` | RMSNorm on queries and keys. |
| `use_flex_attention` | bool | 7 configs | `models/next_concept.py:294` | `false` in all configs. Used only by `forward()`. |
| `ctx_encoder.enabled` | bool | 7 configs | `models/next_concept.py:135-136` | `false` in all configs. |
| `ctx_encoder.n_layer` | int | 7 configs | `models/next_concept.py:154` | Read only when `enabled: true`. |
| `ctx_encoder.vocab_size` | int | 7 configs | `models/next_concept.py:155` | Read only when `enabled: true`. |
| `post_upsample_conv.enabled` | bool | `lm1b-*`, `owt` | `models/next_concept.py:183-184` | Zero-initialized per-level `Conv1d` on the generator input. |
| `post_upsample_conv.kernel_size` | int | `lm1b-*`, `owt` | `models/next_concept.py:186-187` | Must be odd. 3 in both. |
| `tie_heads_to_codebook` | bool | 6 configs | `models/next_concept.py:223`, `:257` | `true`: distance-form head on the frozen codebook. |
| `tie_heads_learn_scale` | bool | `owt`, `tinystories`, `xsum` | `models/next_concept.py:225` | Learned logit scale for the tied head. |
| `tie_heads_bias` | bool | `owt` | `models/next_concept.py:227` | Bias per code on the tied head. |
| `shared_output_head` | bool | `lm1b-*` | `models/next_concept.py:216` | One linear head for all levels. Needs equal sizes. |
| `shared_head_per_level_bias` | bool | `lm1b-*` | `models/next_concept.py:219` | One bias per level on the shared head. |
| `untied_head_bias` | bool | `wmt14` | `models/next_concept.py:220`, `:234` | Bias on untied heads. Default `true`; WMT14 sets `false`. |

## `tokenizer.pre_tokenizer`

| Field | Type | Configs | Read at | Meaning |
|-------|------|---------|---------|---------|
| `name` | enum | all | `tokenization/vqvae_tokenizer.py:15`, `utils/registry.py:4-9` | `hf`, `bpe` or `sudoku`. |
| `tokenizer` | string | all | Not read | |
| `model_id` | string | 6 configs | `tokenization/hf_tokenizer.py:30`, `scripts/eval_shard.py:91`, `scripts/eval_merge.py:116` | Hugging Face tokenizer: `gpt2`, `bert-base-uncased`, `Qwen/Qwen3-0.6B`. |
| `special_tokens.eos_token` | string | 7 configs | `tokenization/hf_tokenizer.py:11-12`; BPE by value at `tokenization/bpe_tokenizer.py:15-17` | EOS token. LM1B uses `[CLS]`, WMT14 `<|im_end|>`. |
| `special_tokens.pad_token` | string | 7 configs | `tokenization/hf_tokenizer.py:13-19`; `tokenization/bpe_tokenizer.py:18` | Pad token. Added as a new token if it does not exist. |
| `vocab_file_path` | path | `tinystories` | `tokenization/bpe_tokenizer.py:22` | Overwritten by `checkpoints.bpe_vocab`. |
| `merges_file_path` | path | `tinystories` | `tokenization/bpe_tokenizer.py:24` | Overwritten by `checkpoints.bpe_merges`. |

## `tokenizer.vqvae`

All fields go to `VQVAE(**config.vqvae)` (`tokenization/vqvae_tokenizer.py:17`, `models/vqvae.py:11-33`).

| Field | Type | Read at | Meaning |
|-------|------|---------|---------|
| `in_vocab_size` | int | `models/vae.py:126`, `:233` | Pre-tokenizer vocabulary size (embedding and output layer). |
| `context_length` | int | `models/vqvae.py:39`, `models/vae.py:127` | L. Also the size of the position table. |
| `embed_dim` | int | `models/vqvae.py:36` | D, the size of codebook vectors: 384, or 256 for Sudoku. |
| `n_layers` | int | `models/vae.py:132`, `:224-227` | Attention head count in the encoder and decoder blocks. Not a depth. |
| `compression_factor` | int | `models/vae.py:147-157`, `:236-245` | 1 in all configs: no extra length compression. |
| `dropout` | float | `models/vae.py:128`, `:220` | Dropout in the tokenizer. Inactive in eval mode. |
| `pre_quant_groupnorm` | int | `models/vqvae.py:60-67` | GroupNorm groups before quantization, no affine. |
| `pre_quant_dropout` | float | `models/vqvae.py:69-71` | Dropout before quantization. Inactive in eval mode. |
| `pre_quant_norm_type` | enum | `models/vqvae.py:62` | `group` (Sudoku sets it, default too) or `layer`. |
| `use_qk_norm` | bool | `models/vae.py:40-44` | QK norm in tokenizer attention. |
| `enc_positional` | bool | `models/vqvae.py:47`, `models/vae.py:127` | Learned encoder positions. `false` for LM1B. |
| `dec_positional` | bool | `models/vqvae.py:56`, `models/vae.py:219` | Learned decoder positions. `true` for LM1B. |
| `dec_layers` | int | `models/vae.py:228-231` | Decoder blocks. 4 for LM1B, 1 otherwise. |
| `vqvae_vocab_size` | list of int | `models/quant.py:290-296` | Codebook size per level. Length must be K. |
| `checkpoint_path` | path | `tokenization/vqvae_tokenizer.py:18` | Set again by `scripts/eval_shard.py:45`. |

## `tokenizer.vqvae.vector_quantizer_config`

All fields except `clss` go to `MultiscaleResidualVectorQuantizer(...)` (`models/quant.py:226`, `:229-250`).

| Field | Type | Read at | Meaning |
|-------|------|---------|---------|
| `clss` | string | `models/vqvae.py:50`, `utils/registry.py:11-13` | `multiscale_residual_vector_quantizer`. |
| `decay` | float | `models/quant.py:256-261` | EMA decay. 0.99, or 0.95 for TinyStories. Training only. |
| `epsilon` | float | `models/quant.py:254`, `:496` | Laplace smoothing in the EMA. Training only. |
| `commitment_cost` | float | `models/quant.py:253`, `:519` | Weight of the commitment term. Training only. |
| `learned_l1_sampling` | bool | `models/quant.py:271`, `:311-328` | Learned level-1 resample. `true` in all configs. |
| `learned_l1_cascade` | bool | `models/quant.py:272`, `:317-322` | Split the level-1 stride into factors. `false` for LM1B. |
| `learned_l1_cascade_base` | int | `models/quant.py:273` | Largest factor: 4. |
| `learned_l1_cascade_norm` | bool | `models/quant.py:274` | GroupNorm between cascade steps. `true` XSum, WMT14. |
| `learned_l1_cascade_norm_final` | bool | `models/quant.py:275` | GroupNorm after the last down step. `true` XSum, WMT14. |
| `learned_all_sampling` | bool | `models/quant.py:270`, `:330-342` | Learned refinement at levels 2 to K−1. `false` in all configs. |
| `ema_scale_by_world_size` | bool | `models/quant.py:255`, `:416-422` | Scale EMA decay by GPU count. `false`. |
| `quant_resi.enabled` | bool | `models/quant.py:109` | Use Phi. `true`. |
| `quant_resi.ratio` | float | `models/quant.py:112` | r in `(1 − r) h + r Conv1d(h)`. 0.5. |
| `quant_resi.share_mode` | int | `models/quant.py:113`, `:139-143` | 0: one Phi per level. |
| `quant_resi.learnable_ratio` | bool | `models/quant.py:114` | Learn r. `false`. |
| `aux.fine_drop.prob` | float | `models/quant.py:264` | Probability of fine-level dropout. 0.5, or 1.0 for LM1B. Training only. |
| `aux.fine_drop.min_keep` | int | `models/quant.py:265`, `:526` | Lowest kept level, 1-based. 1. |
| `aux.fine_drop.dual` | bool | Not read | LM1B only. |
| `aux.fine_drop.w_full` | float | Not read | LM1B only, 0.7. |
| `aux.fine_drop.w_trunc` | float | Not read | LM1B only, 0.3. |
| `aux.code_corruption.enabled` | bool | Not read | LM1B only. |
| `aux.code_corruption.mode` | string | Not read | LM1B only, `uniform`. |
| `aux.code_corruption.random_rate` | bool | Not read | LM1B only. |
| `aux.code_corruption.low` | float | Not read | LM1B only, 0.0. |
| `aux.code_corruption.high` | float | Not read | LM1B only, 1.0. |
| `aux.code_corruption.skip_level0` | bool | Not read | LM1B only. |
| `levels.use_manual_levels` | bool | `models/quant.py:285` | `true` in all configs. |
| `levels.manual_levels` | list of int | `models/quant.py:286` | The level schedule ℓ_1 … ℓ_K. |

## `generation`

All fields are read in `scripts/eval_shard.py`. A command-line flag overrides each one ([CLI reference](cli.md)).

| Field | Type | Read at | Default if absent | Meaning |
|-------|------|---------|-------------------|---------|
| `temperature` | float | `:34` | 1.0 | Sampling temperature. 0 gives argmax. |
| `top_k` | int | `:35` | 1 | Keep the k largest logits. 0 turns it off. |
| `top_p` | float | `:36` | 0.0 | Nucleus threshold. 1.0 keeps all. |
| `cfg_scale` | float | `:37` | 0.0 | Guidance weight at the last level. |
| `frequency_penalty` | float | `:38` | 0.0 | Token frequency penalty in `pick_tokens()`. |
| `frequency_deficit` | float | `:39` | 0.0 | Bonus for tokens under their allowance. |
| `adjacency_penalty` | string `"a,w,d"` | `:40`, `:80`, `:105-107` | empty | Penalty for recent tokens. |
| `frequency_allowance` | `file:key.key` | `:69`, `:59-66` | none | Token unigram in `generator.pt` at `inference.token_unigram`. |
| `level0_unigram` | `file:key.key` | `:71` | none | Level-1 code table at `inference.level0_unigram`. |
| `cut_at_eos` | bool | `:127` | `true` | Cut the token row at the first EOS. OWT sets `false`. |

## `eval`

| Field | Type | Read at | Meaning |
|-------|------|---------|---------|
| `split` | string | `scripts/eval_shard.py:41` | Data split. Default `test`. |
| `n_docs` | int | `scripts/eval_shard.py:42` | Inputs to evaluate: 11334 XSum, 3003 WMT14, 2000 Sudoku. |
| `n_samples` | int | `scripts/eval_shard.py:84` | Samples per seed: 256. |
| `seeds` | list of int | `scripts/eval_shard.py:83` | `42 0 1 2 3`. Split across shards. |
| `batch_size` | int | `scripts/eval_shard.py:43` | Batch size per shard. |
| `n_shards` | int | Not read | `evaluate.py:20` uses the GPU count. |
| `metric` | enum | `scripts/eval_merge.py:154` | `exact`, `rouge`, `bleu`, `genppl`, `genppl_flm`. |
| `ref_model` | string | `scripts/eval_merge.py:86`, `:117` | Scoring model: `gpt2-large`. |

## Values per config

| Config | K | `manual_levels` | P | Codebook | Generator `n_layer`/`n_embd`/`n_head` |
|--------|---|-----------------|---|----------|---------------------------------------|
| `lm1b-small` | 12 | `1 8 16 24 32 40 52 64 76 92 108 128` | 1 | 8192 | 12 / 768 / 12 |
| `tinystories` | 12 | `1 8 16 24 32 40 52 64 76 92 108 128` | 1 | 4096 | 12 / 768 / 12 |
| `owt` | 32 | k² for k = 1 … 32 | 1 | 8192 | 16 / 1024 / 16 |
| `xsum` | 9 | `4 6 10 14 20 26 32 40 48` | 512 | 4096 | 12 / 768 / 12 |
| `wmt14` | 11 | `1 4 8 14 20 26 32 40 48 56 64` | 96 | 4096 | 12 / 768 / 12 |
| `sudoku-*` | 12 | `1 2 4 6 9 13 18 26 37 50 67 89` | 91 | 16 | 8 / 512 / 8 |

The values for `lm1b-medium` and `lm1b-large` are in their files. On 2026-11-21 their generators had 316.11M and 724.29M non-embedding parameters (random weights, counted by the constructor).
