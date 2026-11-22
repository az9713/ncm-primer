# Project 4: show the coarse-to-fine trajectory

**Goal:** Make an HTML page that shows one sample after each level, from level 1 to level K, as in `assets/coarse_to_fine.gif`.

**Effort:** about 20 lines. **Status:** run on 2026-11-21/22 (Sudoku hard puzzle 0 and WMT14 source 1218). Full results: [experiment report](../../experiments/index.html).


## Result (run 2026-11-21/22)

- **What it measured:** the output after each level, compared with paper Fig. 20 (Sudoku) and Table 12 (WMT14).
- **Result:** WMT14 source 1218 reproduces Table 12 exactly: 11 of 11 per-level BLEU values and all texts. Only method B (`partial_decodes()`) gives this.
- **Result:** Sudoku puzzle 0 reaches the Fig. 20 final board. Intermediate correct-cell counts match at 5 of 12 levels in the default precision and 9 of 12 in bf16.
- **Unexpected:** numeric precision changes intermediate Sudoku boards. bf16 on all 2,000 hard puzzles gives 86.35 %, the default 86.25 % (Table 3: 86.1 %).
- **Lesson:** the paper decodes intermediate levels with full-length partial sums. A viewer page is in `experiments/p04_coarse_to_fine/trajectory.html`.

## Background

There are two ways to decode a level. They answer two different questions.

| Method | Function | Length at level k | Shows |
|--------|----------|-------------------|-------|
| A | `tk.decode_multiscales(codes)` (`tokenization/vqvae_tokenizer.py:75-86`) | ℓ_k tokens | A short version of the output at the length of the level |
| B | `partial_decodes(tk.vqvae, codes)` ([shared setup](index.md)) | L tokens | The full-length output if generation stopped after level k |

The repository does not say which method made the GIF.

## What exists and what you write

- Exists: `generate()`, `decode_multiscales()`.
- You write: `partial_decodes()` (10 lines, given in the guide) and the HTML output.

## Steps

1. Load a model with `load()` and generate codes as in [Use NCM from Python](../guides/use-ncm-from-python.md).
2. Decode each level with both methods.

   ```python
   per_level_a = tk.decode_multiscales(codes, skip_special_tokens=True)       # list[B][K] strings
   per_level_b = [[tk.pre_tokenizer.decode(r, skip_special_tokens=True) for r in t.tolist()]
                  for t in partial_decodes(tk.vqvae, codes)]                 # list[K][B] strings
   ```

3. Write a table for one sample.

   ```python
   import html
   levels = tk.vqvae.vq.levels
   rows = "".join(f"<tr><td>{k+1}</td><td>{l}</td><td>{html.escape(per_level_a[0][k])}</td>"
                  f"<td>{html.escape(per_level_b[k][0])}</td></tr>" for k, l in enumerate(levels))
   open("trajectory.html", "w", encoding="utf-8").write(
       f"<table border=1><tr><th>level</th><th>l_k</th><th>method A</th><th>method B</th></tr>{rows}</table>")
   ```

## Example output (Sudoku hard, first puzzle, 2026-11-21)

Method A, levels 1 to 4 and 12:

```text
6
86
8886
224666
...
921647385|764835291|583912764|452369178|179458623|836721549|217584936|698173452|345296817
```

## Cost and hardware

One `generate()` call and K decoder calls. For Sudoku hard, one batch of 128 took 3.73 s on an RTX 3050 Laptop GPU.

## How to measure the result

- For text: the token edit distance between level k and level K, for each k. Plot it against k.
- For text: the level at which the first sentence stops changing.
- Compare methods A and B at the same level.
