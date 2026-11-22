# Use NCM from Python

Use this guide to load a released model in your own script, generate codes, and decode them to text. The [projects](../projects/index.md) use the functions on this page.

Prerequisites: the packages in [Prerequisites](../getting-started/prerequisites.md), one downloaded model (see [Run an evaluation](run-an-evaluation.md)).

Status: `load()`, the `generate()` call, the decode in step 5 and `partial_decodes()` were run on 2026-11-21 with the `sudoku-hard` checkpoint on a GPU. The parts for text configs (unigram table, `to_text()` on text) were not run.

> **Warning:** `torch.load(..., weights_only=False)` unpickles Python objects, and a pickle can run code. `scripts/eval_shard.py:52` uses the same call. Load only checkpoint files from a source that you trust.

## Steps

1. Put the repository root on the import path. Set `PYTHONDONTWRITEBYTECODE=1` if you do not want `__pycache__` folders in the repository.

   ```python
   import os, sys
   NCM = "/path/to/ncm"          # the repository root
   sys.path.insert(0, NCM)
   ```

2. Load the tokenizer and the generator. This function copies `scripts/eval_shard.py:45-54`.

   ```python
   import torch
   from omegaconf import OmegaConf
   from models.next_concept import NextConceptTransformer
   from tokenization.vqvae_tokenizer import VQVAETokenizer

   def load(config_path, ck):
       cfg = OmegaConf.load(config_path)
       cfg.tokenizer.vqvae.checkpoint_path = os.path.join(ck, cfg.checkpoints.tokenizer)
       for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
           if cfg.checkpoints.get(name):
               cfg.tokenizer.pre_tokenizer[key] = os.path.join(ck, cfg.checkpoints[name])
       tk = VQVAETokenizer.from_config(cfg.tokenizer)
       tk.vqvae = tk.vqvae.cuda().eval()
       net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model).cuda()
       sd = torch.load(os.path.join(ck, cfg.checkpoints.generator), map_location="cpu", weights_only=False)
       net.load_state_dict(sd["model"], strict=True)
       return cfg, tk, net.eval(), sd

   cfg, tk, net, sd = load(f"{NCM}/config/eval/sudoku-hard.yaml", f"{NCM}/artifacts/sudoku/hard")
   ```

   The Sudoku `generator.pt` has only the key `model`. For configs with `level0_unigram` or `frequency_allowance`, `sd["inference"]` holds the tables (`scripts/eval_shard.py:59-66`).

3. Build the prefix and the level-1 codes. The method depends on the level-1 mode.

   ```python
   P = int(cfg.model.prefix_len)
   EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
   B = 8

   # Conditional (xsum, wmt14, sudoku): P tokens per row, padded on the right with PAD.
   # For WMT14 the inputs are strings; tokenize them with AutoTokenizer first (eval_shard.py:91, :95).
   # prefix = torch.tensor([(ids[:P] + [PAD] * P)[:P] for ids in rows], device="cuda")

   # Unconditional (lm1b, tinystories): one EOS token per row (eval_shard.py:157).
   prefix = torch.full((B, 1), EOS, device="cuda")

   # Level 1 from the unigram table (lm1b, tinystories), as eval_shard.py:156.
   p = sd["inference"]["level0_unigram"].float().cpu()
   l1 = torch.multinomial(p, B * tk.vqvae.vq.levels[0], replacement=True).view(B, -1).cuda()
   # For xsum, wmt14 and sudoku, use l1 = None.
   ```

4. Generate the codes.

   ```python
   with torch.no_grad():
       codes = net.generate(l1, prefix, temperature=1.0, top_k=0, top_p=1.0)
   # codes.shape == (B, sum(tk.vqvae.vq.levels))
   ```

   Use the config values `cfg.generation.temperature`, `top_k`, `top_p` to match the eval. For Sudoku, use `temperature=0.0`.

5. Decode the codes to tokens and text. This copies the eval path without penalties (`scripts/eval_shard.py:130-134`, `:143-144`).

   ```python
   with torch.no_grad():
       _, z_levels = tk.vqvae.vq.idx_to_zhats(codes, return_zlevels=True)
       ids = tk.vqvae.decoder(z_levels[-1]).argmax(-1).tolist()

   def to_text(row):
       if EOS is not None and EOS in row and cfg.generation.get("cut_at_eos", True):
           row = row[:row.index(EOS)]
       return tk.pre_tokenizer.decode([t for t in row if t not in (EOS, PAD)], skip_special_tokens=True)

   texts = [to_text(r) for r in ids]
   ```

   The LM1B and XSum evals use `pick_tokens()` with penalties in place of `argmax`. Your text can thus differ from the eval output for these configs.

6. Decode each level. There are two methods.

   ```python
   # Method A: l_k tokens per level (tokenization/vqvae_tokenizer.py:75-86).
   per_level = tk.decode_multiscales(codes, skip_special_tokens=True)   # per_level[b][k] is a string

   # Method B: L tokens per level, from the partial sum of levels 1..k.
   @torch.no_grad()
   def partial_decodes(vqvae, codes):
       vq = vqvae.vq
       L, K = vq.levels[-1], len(vq.levels)
       z_hat = torch.zeros(codes.shape[0], L, vq.embed_dim, device=codes.device)
       out, a = [], 0
       for k, l in enumerate(vq.levels):
           e = vq.embeddings[k](codes[:, a:a + l])
           q = vq._up_to_full(e, k, L) if k < K - 1 else e
           z_hat = z_hat + vq._apply_quant_resi(q, k)   # a new tensor: no aliasing
           out.append(vqvae.decoder(z_hat).argmax(-1))
           a += l
       return out                                        # K tensors of shape (B, L)
   ```

   Method B does not use `idx_to_zhats()`'s `z_hats` list, because all its entries are the same tensor (`models/quant.py:651-652`).

## Verification

- On 2026-11-21, `partial_decodes(...)[-1]` was equal to `decoder(z_levels[-1]).argmax(-1)` for 4 Sudoku boards (`torch.equal` gave `True`).
- `codes.shape[1]` must equal `sum(levels)`: 641 (LM1B, TinyStories), 200 (XSum), 313 (WMT14), 322 (Sudoku).

## Troubleshooting for this task

| Symptom | Cause | Fix |
|---------|-------|-----|
| `RuntimeError: selected index k out of range` | `top_k` is larger than the codebook size (16 for Sudoku). The default of `generate()` is 50. | Pass `top_k=0` or a value of 16 or less. |
| `AssertionError: query_level0 generation supports the eager path only` | `use_kv_cache=True` with `xsum`, `sudoku-*` or `lm1b-*`. | Use `use_kv_cache=False`. |
| `ValueError: too many values to unpack (expected 3)` | You called `vqvae.reconstruct()` or `tk.reconstruct()`, which are broken (`models/vqvae.py:116`). | Use `tk.vqvae.encode()` and the decode steps above. |
| `RuntimeError: Calculated padded input size per channel ...` | The input to `encode()` is not `context_length` tokens long. | Pad to `context_length`. |
