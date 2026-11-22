# The generator

`NextConceptTransformer` predicts the codes of one level per pass. All codes of a level come from the same pass, in parallel.

This page tracks `models/next_concept.py` and `models/common.py`.

## Inputs and outputs

The constructor is `NextConceptTransformer(vqvae_local, config)` (`models/next_concept.py:67-70`). `vqvae_local` is the loaded `VQVAE`. `config` is the `model:` section of a config.

`generate(idx, prefix_tokens, temperature=1.0, top_k=50, top_p=0.9, use_kv_cache=False, cfg_scale=0.0, cfg_null_eos=50256)` (`:521-522`) returns a tensor of shape (B, ℓ_1 + … + ℓ_K) with the codes of all levels, level 1 first (`:699-703`).

| Argument | Meaning |
|----------|---------|
| `idx` | Level-1 codes, shape (B, ℓ_1), or `None`. With `None`, the model makes level 1 itself. |
| `prefix_tokens` | Pre-tokenizer ids, shape (B, P). |
| `temperature`, `top_k`, `top_p` | Sampler settings, see [Sampling](#sampling). |
| `use_kv_cache` | Compute the prefix keys and values once. Forbidden with `query_level0` (`:523-524`). |
| `cfg_scale` | Guidance weight. Needs `prefix_pad_token_id` (`:531-532`). |
| `cfg_null_eos` | Token at position 0 of the null prefix (`:535`). |

## The model stack

| Part | Lines | Note |
|------|-------|------|
| Input projection `wte` | `:131-133` | A linear map from the tokenizer size D (384 or 256) to `n_embd`. |
| Prefix embedding | `:546-560` | With `raw_token_prefix: true` (all configs), the prefix goes through the tokenizer encoder embedding `wte` only. |
| Positions | `:95-127` | With `use_rope: true` (all configs), RoPE restarts at 0 for each level (`:110-117`). The prefix has positions 0 to P−1. |
| Level embedding `wle` | `:197-200` | One learned vector per level. |
| Post-upsample convolution | `:183-195` | A zero-initialized per-level `Conv1d` on the input. LM1B and OWT only. |
| Blocks | `:202-206`, `models/common.py:167-213` | Pre-norm transformer blocks with QK norm (`use_qk_norm: true`). |
| Output heads | `:208-237` | One per predicted level. Three kinds, see below. |

| Head kind | Config | Lines |
|-----------|--------|-------|
| Tied to the codebook: logits are `(2 z·E − ‖E‖²) / mean‖E‖²` times a learned scale | `tie_heads_to_codebook: true` (OWT, TinyStories, XSum) | `:9-34`, `:223-232` |
| One shared linear head with a bias for each level | `shared_output_head: true` (LM1B) | `:48-62`, `:216-222` |
| One linear head for each level | default (WMT14, Sudoku) | `:233-237` |

## The attention mask

The mask has shape (P + T, P + T). T is the total target length (`:268-292`).

- Prefix rows see the prefix only (`:282-283`).
- Target rows see the full prefix (`:284`).
- With `attn_pattern: block_diagonal` (all configs), a target row sees only the rows of its own level (`:274`).
- With `block_causal`, a target row sees its own level and all earlier levels (`:276`).

In `generate()`, each pass has its own small mask (`:507-510`): prefix rows see the prefix, level rows see everything.

## Level-1 modes

| Mode | Config fields | Configs | Where | Passes |
|------|---------------|---------|-------|--------|
| Unigram table | `query_level0: false` + `generation.level0_unigram` | `tinystories`, `owt` | `scripts/eval_shard.py:71`, `:156` | K−1 |
| Unigram table, learned queries not used | `query_level0: true` + `generation.level0_unigram` | `lm1b-*` | Same; the eval passes `idx`, so `:581` is skipped | K−1 = 11 |
| Learned queries | `query_level0: true`, no unigram | `xsum`, `sudoku-*` | `:239-242`, one pass at `:581-627` | K |
| Prefix head | `predict_level0: true` | `wmt14` | `:249-264`, prefix pass at `:629-641` | K |

With `predict_level0`, the head reads the last ℓ_1 non-pad prefix positions (`_level0_slice`, `:427-435`). For WMT14, ℓ_1 = 1.

## The generation loop

After level 1, the loop at `:660-694` runs K−1 times. At step k (0-based `level_idx`):

1. Look up the codebook vectors of the newest codes (`:664`).
2. Lengthen them to L, apply Phi, add them to z_hat, and pool z_hat to ℓ_{k+1} positions (`vq.get_next_autoregressive_input`, `models/quant.py:697-709`).
3. Project the input, add the level and position embeddings (`:670`).
4. Run one pass over `[prefix ; input]` and apply the head of this level (`:689-691` eager, `:680-688` cached).
5. Sample the ℓ_{k+1} codes (`_sample_level`).

In the eager path, each pass recomputes the prefix (P + ℓ_{k+1} rows). The prefix rows see only the prefix, so their values are the same at each level. The KV-cache path computes the prefix once (`:649-657`) and reuses its keys and values (`:681-682`).

> **Warning:** The KV-cache path does not add the prefix pad mask (`:654`, `:683`). On 2026-11-21, with random weights and a half-padded WMT14 prefix, the cached and eager codes agreed at 49.4% of positions. With an unpadded prefix they agreed at 100%. `scripts/eval_shard.py:142` never sets `use_kv_cache`.

## Sampling

`_sample_level_raw` (`:487-499`):

- `temperature == 0`: argmax.
- Otherwise: divide by temperature. Keep the `top_k` largest logits if `top_k > 0`. Keep the smallest set with cumulative probability above `top_p`. Draw one sample.
- The top-p step always runs. `top_p: 1.0` keeps all tokens. `top_p: 0.0` keeps the top token only.

| Config | temperature | top_k | top_p | Effect |
|--------|-------------|-------|-------|--------|
| `xsum`, `wmt14` | 1.0 | 1 | 0.0 | Top token only (greedy) |
| `sudoku-*` | 0.0 | 0 | 1.0 | Argmax |
| `lm1b-*`, `tinystories`, `owt` | 1.0 | 0 | 1.0 | Full sampling |

> **Warning:** The default `top_k=50` of `generate()` fails on Sudoku, where the vocabulary has 16 codes. The check on 2026-11-21 raised `RuntimeError: selected index k out of range`. Set `top_k` to 16 or less.

## Classifier-free guidance

With `cfg_scale > 0`, `generate()` doubles the batch. The second half has a null prefix: `prefix_pad_id` everywhere, `cfg_null_eos` at position 0 (`:530-540`). At each level, `_sample_level` mixes the two halves (`:477-485`):

```text
t = cfg_scale * level / (K - 1)
logits = (1 + t) * logits_cond - t * logits_null
```

The weight t is 0 at level 1 and `cfg_scale` at level K (`:481-482`). Both halves get the same sampled codes (`:484`).

- Sudoku has no `prefix_pad_token_id`. `cfg_scale > 0` raises `ValueError: cfg_scale needs prefix_pad_id to build the null prefix`.
- For the unconditional configs, P = 1 and the prefix is one EOS token. The null prefix is also one EOS token, so guidance has no effect.

## Training-mode forward

`forward(x, prefix_ids=None, prefix_pad_mask=None)` (`:376-406`) runs all levels in one pass with the full mask. `x` is `[prefix vectors ; target inputs]`. `VQVAE.get_ncm_inputs()` (`models/vqvae.py:121-124`) builds the target inputs and targets for configs without `query_level0`. The forward returns logits padded to the largest vocabulary (`:467-471`). The repository has no loss and no optimizer for it.

Related: [The tokenizer](tokenizer.md), [System design](../architecture/system-design.md).
