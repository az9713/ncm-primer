# Project 6: measure wall-clock time against pass count

**Goal:** Measure the time of each generator pass and the quality after each level. The result is a curve of quality against time, without changes to `generate()`.

**Effort:** about 30 lines. **Status:** the timing code was run on Sudoku hard on 2026-11-21.

## Background

`generate()` always runs all levels. But the first k levels of a run do not depend on the later levels. So the output of a run stopped after level k equals `partial_decodes(...)[k-1]` of the full run. Its cost is the sum of the first k pass times.

The KV cache (`use_kv_cache=True`) computes the prefix once (`models/next_concept.py:649-657`). It is allowed only for configs without `query_level0`: `tinystories` and `wmt14` of the released models (`:523-524`). It ignores the prefix pad mask (`:654`, `:683`).

## What exists and what you write

- Exists: `generate()` with and without KV cache.
- You write: CUDA-event hooks and a timing loop.

## Steps

1. Load a model and build a batch as in [Use NCM from Python](../guides/use-ncm-from-python.md).
2. Time the whole call. Synchronize before you read the clock.

   ```python
   import time, torch
   def time_generate(net, l1, prefix, n=5, **kw):
       with torch.no_grad():
           net.generate(l1, prefix, **kw)                  # warm-up
           torch.cuda.synchronize(); t = time.perf_counter()
           for _ in range(n):
               net.generate(l1, prefix, **kw)
           torch.cuda.synchronize()
       return (time.perf_counter() - t) / n
   ```

3. Time each pass. Each pass starts at the first block and ends at `ln_f`.

   ```python
   starts, ends = [], []
   def mark(lst):
       ev = torch.cuda.Event(enable_timing=True); ev.record(); lst.append(ev)
   h1 = net.transformer.blocks[0].register_forward_pre_hook(lambda m, a: mark(starts))
   h2 = net.transformer.ln_f.register_forward_hook(lambda m, a, o: mark(ends))
   with torch.no_grad():
       codes = net.generate(l1, prefix, temperature=0.0)
   torch.cuda.synchronize(); h1.remove(); h2.remove()
   per_pass_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
   ```

4. Compute the quality after each level with `partial_decodes(tk.vqvae, codes)` and the metric of the task.
5. For `tinystories` and `wmt14`, repeat step 2 with `use_kv_cache=True`. Use unpadded prefixes for WMT14.
6. Repeat for batch sizes 1, 16 and 128.

## Measured example (Sudoku hard, 2026-11-21)

NVIDIA GeForce RTX 3050 Laptop GPU, batch 128, argmax, eager path.

| Item | Value |
|------|-------|
| Full `generate()` (mean of 5) | 3.73 s |
| Pass times, levels 1 to 12 (ms) | 236.69, 238.17, 243.91, 250.32, 255.58, 267.85, 278.54, 296.88, 323.40, 372.81, 410.32, 478.58 |

Pass 12 (91 + 89 rows) took 478.58 / 236.69 = 2.02 times as long as pass 1 (91 + 1 rows).

## Cost and hardware

Minutes on one GPU. Pass times depend on the GPU, the batch size and P.

## How to measure the result

- Plot quality against cumulative time, one point per level.
- Report the time per output token: total time / (B × L).
- Report the KV-cache speedup for `tinystories` and `wmt14`.
