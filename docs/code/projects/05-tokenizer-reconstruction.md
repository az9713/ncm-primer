# Project 5: measure tokenizer reconstruction per level

**Goal:** Measure how many tokens the frozen tokenizer gets right when the decoder sees only levels 1 to k. Use your own text.

**Effort:** about 20 lines. **Status:** the code was run with random weights on the Sudoku config (2026-11-21). The released tokenizers were not tested.

## Background

In training mode, fine-level dropout gives the decoder z_hat cut at a random level (`models/quant.py:525-530`). So the decoder learned to decode full-length partial sums. `z_to_idx(z, return_zhats=True)` returns K full-length partial sums. They are separate tensors (`models/quant.py:609-611`).

> **Warning:** Do not use `VQVAE.reconstruct()` or `VQVAETokenizer.reconstruct()`. They raise `ValueError: too many values to unpack (expected 3)` (`models/vqvae.py:116`).

## What exists and what you write

- Exists: `encode_z()`, `z_to_idx(..., return_zhats=True)`, `decoder`.
- You write: tokenization of your text, padding to L, the accuracy loop.

## Steps

1. Load a tokenizer. The generator is not needed, but `load()` builds it. You can also call `VQVAETokenizer.from_config(cfg.tokenizer)` directly.
2. Tokenize your text and cut or pad each row to `context_length` tokens. `encode()` accepts only exactly L tokens.

   ```python
   import torch
   L, PAD = tk.vqvae.context_length, tk.pre_tokenizer.pad_token_id
   ids = tk.pre_tokenizer.encode(open("my_text.txt", encoding="utf-8").read())
   rows = [ids[i:i + L] for i in range(0, len(ids) - L + 1, L)]      # full rows only
   x = torch.tensor(rows, device="cuda")
   ```

3. Compute the accuracy after each level.

   ```python
   with torch.no_grad():
       _, zh = tk.vqvae.vq.z_to_idx(tk.vqvae.encode_z(x), return_zhats=True)
       acc = [(tk.vqvae.decoder(z).argmax(-1) == x).float().mean().item() for z in zh]
   for k, (l, a) in enumerate(zip(tk.vqvae.vq.levels, acc)):
       print(f"level {k+1:2d}  l_k={l:4d}  token accuracy {a:.4f}")
   ```

4. Repeat with text from a different domain, for example code or another language. Compare the curves.

## Cost and hardware

Encoding and K decoder calls, no generator. The time was not measured. The LM1B tokenizer has 111.40M parameters; the TinyStories tokenizer 36.43M.

## How to measure the result

- Plot token accuracy against ℓ_k, one line for each domain.
- Report the accuracy at level K. If it is below 1.0, the tokenizer loses information even with all levels.
- Exclude pad tokens from the accuracy when you pad rows.
