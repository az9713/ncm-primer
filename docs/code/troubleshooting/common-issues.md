# Common issues

Each item gives the symptom, the cause and the fix. Items marked "checked" were reproduced on 2026-11-21. The other items come from reading the code.

This page tracks `evaluate.py`, `scripts/*.py` and `models/*.py`.

## Setup and paths

### The README checkpoint path does not exist

- **Symptom:** `FileNotFoundError` for `artifacts/sudoku-hard/tokenizer.pt` or `artifacts/lm1b-small/generator.pt`.
- **Cause:** The README uses `artifacts/<task>`. The Hugging Face layout is `sudoku/hard/`, `lm1b/small/` (listed 2026-11-21).
- **Fix:** Use `--checkpoint-dir artifacts/sudoku/hard --data-dir data/sudoku/hard`, or `--checkpoint-dir artifacts/lm1b/small`. Checked for Sudoku hard.

### The OWT evaluation cannot start

- **Symptom:** No `generator.pt` or `tokenizer.pt` in `artifacts/owt/`.
- **Cause:** On 2026-11-21 the Hugging Face folder `owt/` holds only `TODO.md`.
- **Fix:** None. Wait for the release.

### The data download is 22.741 GB

- **Symptom:** `hf download hazyresearch/ncm-datasets ...` from the README downloads training streams.
- **Cause:** The dataset repository holds training data too, for example 18.144 GB for OWT.
- **Fix:** Add `--include "xsum/*"` (or the folder you need). Unconditional configs need no data.

### Output files go to two different folders

- **Symptom:** `shard_0.log` is in one folder, `shard_0.pkl` in another, or the merge prints `no shard files`.
- **Cause:** Shards run with `cwd` set to the repository root (`evaluate.py:35`). The log opens relative to your working directory (`evaluate.py:31`).
- **Fix:** Run from the repository root, or pass absolute paths to `--out`, `--checkpoint-dir` and `--data-dir`.

### No GPU

- **Symptom:** `no GPU visible; pass --gpus`.
- **Cause:** `torch.cuda.device_count()` returned 0 (`evaluate.py:20-22`). The shards also call `.cuda()` (`scripts/eval_shard.py:50-51`).
- **Fix:** Install a CUDA build of PyTorch. The scripts have no CPU mode.

### Missing `omegaconf` or `hydra`

- **Symptom:** `ModuleNotFoundError: No module named 'omegaconf'`. Checked.
- **Fix:** `pip install -r requirements.txt`.

## Generation

### `top_k` larger than the codebook

- **Symptom:** `RuntimeError: selected index k out of range`. Checked with `sudoku-hard`.
- **Cause:** `torch.topk` with k = 50 (default of `generate()`) over 16 codes (`models/next_concept.py:492`).
- **Fix:** Pass `top_k=0` or a value up to 16 for Sudoku.

### Guidance on Sudoku

- **Symptom:** `ValueError: cfg_scale needs prefix_pad_id to build the null prefix`. Checked.
- **Cause:** The Sudoku configs have no `prefix_pad_token_id` (`models/next_concept.py:531-532`).
- **Fix:** Do not use `--cfg` with Sudoku.

### Guidance has no effect on unconditional configs

- **Symptom:** The same scores with `--cfg 0` and `--cfg 1`.
- **Cause:** With P = 1, the real prefix and the null prefix are both one EOS token (`scripts/eval_shard.py:157`, `models/next_concept.py:534-535`).
- **Fix:** Use guidance only with XSum and WMT14.

### KV cache refused

- **Symptom:** `AssertionError: query_level0 generation supports the eager path only`.
- **Cause:** `use_kv_cache=True` with `xsum`, `sudoku-*` or `lm1b-*` (`models/next_concept.py:523-524`).
- **Fix:** Use the eager path. Of the released models, only `tinystories` and `wmt14` allow the KV cache.

### KV cache gives different output on WMT14

- **Symptom:** Cached and eager outputs differ for inputs shorter than 96 tokens. Checked with random weights: 49.4% of codes agreed with a half-padded prefix, 100% with no padding.
- **Cause:** The cached path does not add the pad mask (`models/next_concept.py:654`, `:683`).
- **Fix:** Use the eager path for padded prefixes.

### Sampled runs are not repeatable on conditional tasks

- **Symptom:** Two runs of XSum with `--top_k 0 --top_p 0.9` give different scores.
- **Cause:** The conditional path sets no seed (`scripts/eval_shard.py:164-169`).
- **Fix:** Call `torch.manual_seed()` in your own script before each `generate()` call.

## Known code defects

### `reconstruct()` fails

- **Symptom:** `ValueError: too many values to unpack (expected 3)`. Checked.
- **Cause:** `models/vqvae.py:116` unpacks 3 values; `vq.forward` returns 5 (`models/quant.py:543-549`). `VQVAETokenizer.reconstruct()` and `reconstruct_text_from_ids()` call it.
- **Fix:** Use `encode()` and then `decode()`, or the method in [Project 5](../projects/05-tokenizer-reconstruction.md).

### All partial sums from `idx_to_zhats()` are equal

- **Symptom:** Decodes of `z_hats[0]` and `z_hats[-1]` are the same. Checked: `zh[0] is zh[-1]` is `True`.
- **Cause:** In-place `z_hat.add_()` and `append(z_hat)` of one tensor (`models/quant.py:651-652`).
- **Fix:** Use `partial_decodes()` from [Use NCM from Python](../guides/use-ncm-from-python.md), or `z_to_idx(..., return_zhats=True)`.

### Wrong input length to `encode()`

- **Symptom:** `RuntimeError: Calculated padded input size per channel: (1). Kernel size: (2). Kernel size can't be greater than actual input size`. Checked with 100 tokens on TinyStories.
- **Cause:** The learned level-1 resampler has a fixed stride L/ℓ_1.
- **Fix:** Pad each row to `context_length` tokens with the pad id.

### The `__main__` blocks fail

- **Symptom:** `python models/quant.py` raises `AttributeError: 'list' object has no attribute 'use_manual_levels'`. Checked.
- **Cause:** `models/quant.py:730-732` passes an int vocabulary size and a list of levels. `models/vqvae.py:146-158` passes an int vocabulary size too.
- **Fix:** Use the configs in `config/eval/`. The blocks are not needed for evaluation.

### `eval.n_shards` has no effect

- **Cause:** `evaluate.py:20` uses the GPU count. No file reads `n_shards`.
- **Fix:** Use `--gpus`.
