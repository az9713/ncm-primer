# Project 7: trace Sudoku rule violations per level

**Goal:** Count the Sudoku rule violations of the full-length partial decode after each of the 12 levels. Find the level where the board becomes valid.

**Effort:** about 40 lines. **Status:** run on 2026-11-21 on all 2,000 puzzles of each difficulty. Full results: [experiment report](../../experiments/index.html).


## Result (run 2026-11-21/22)

- **What it measured:** rule violations, correct cells and changed clues after each level, and the solved rate.
- **Result:** solved 99.40 / 96.45 / 86.25 % for easy / medium / hard (Table 3: 99.5 / 96.5 / 86.1). The differences are 1 to 3 boards of 2,000.
- **Result:** no board is exact before level 12. On hard, mean violations fall from 34.29 at level 11 to 0.47 at level 12.
- **Unexpected:** violations can rise between coarse levels (hard: 85.89 → 91.12 at level 6) while correct cells rise. 23 level-1 boards with zero violations contain no digits.
- **Lesson:** the last level, with one code per cell, does almost all of the constraint repair.

## Background

A board is 89 tokens: 81 cells and 8 `|` separators. The puzzle prefix is 91 tokens: `<BOS>`, the 89-token board with `_` for blanks, `<BOS>`. The 2,000 hard puzzles are lists of ints. The solutions are lists of 89 ints. The eval counts a board as valid when all 9 rows, 9 columns and 9 boxes hold `1` to `9` once (`scripts/eval_merge.py:62-69`).

## What exists and what you write

- Exists: the checkpoint, the data, `generate()`.
- You write: `partial_decodes()`, a violation counter, a loop.

## Steps

1. Download Sudoku hard (0.163 GB) as in the [Quickstart](../getting-started/quickstart.md).
2. Load the model and 128 puzzles.

   ```python
   import pickle, torch
   from ncm_helpers import load, partial_decodes
   cfg, tk, net, sd = load("config/eval/sudoku-hard.yaml", "artifacts/sudoku/hard")
   items = pickle.load(open("data/sudoku/hard/puzzles.pickle", "rb"))
   sols = pickle.load(open("data/sudoku/hard/solutions.pickle", "rb"))
   B = 128
   x = torch.tensor([list(it)[:91] for it in items[:B]], device="cuda")
   with torch.no_grad():
       codes = net.generate(None, x, temperature=0.0, top_k=0, top_p=1.0)
   ```

3. Count violations. A violation is a repeated digit in a row, column or box.

   ```python
   def violations(text):
       g = text.replace("|", "")
       if len(g) != 81:
           return None
       rows = [g[i*9:(i+1)*9] for i in range(9)]
       cols = ["".join(r[j] for r in rows) for j in range(9)]
       boxes = ["".join(rows[r][c] for r in range(br, br+3) for c in range(bc, bc+3))
                for br in (0, 3, 6) for bc in (0, 3, 6)]
       return sum(len([ch for ch in u if ch != "_"]) - len(set(ch for ch in u if ch != "_"))
                  for u in rows + cols + boxes)

   for k, p in enumerate(partial_decodes(tk.vqvae, codes)):
       texts = [tk.pre_tokenizer.decode(r) for r in p.tolist()]
       v = [violations(t) for t in texts]
       ok = [t for t in v if t is not None]
       exact = sum(t == tk.pre_tokenizer.decode(s) for t, s in zip(texts, sols[:B]))
       print(f"level {k+1:2d}  mean violations {sum(ok)/max(len(ok),1):6.2f}  exact {exact}/{B}")
   ```

4. Add a count of changed clues: the cells where the puzzle has a digit and the decode has a different digit.

## Measured result (2026-11-21)

First 128 hard puzzles, argmax, RTX 3050 Laptop GPU. All 128 decodes parsed to 81 cells at every level.

| Level | ℓ_k | Mean violations per board | Exact |
|-------|-----|---------------------------|-------|
| 1 | 1 | 95.19 | 0/128 |
| 4 | 6 | 85.02 | 0/128 |
| 8 | 26 | 73.32 | 0/128 |
| 10 | 50 | 44.47 | 0/128 |
| 11 | 67 | 35.20 | 0/128 |
| 12 | 89 | 0.55 | 109/128 |

From level 11 to level 12, the mean dropped from 35.20 to 0.55. No board was exact before level 12. The full run on 2,000 puzzles gave 1,725 exact (86.25%).

## Cost and hardware

One batch of 128 took 3.73 s for `generate()` on the RTX 3050 Laptop GPU.

## How to measure the result

- Plot the mean violations against the level, with one line for each difficulty (easy, medium, hard).
- Report the number of changed clues per level.
- For the 19 unsolved boards in the measured batch, find the level where their violation count stops falling.
