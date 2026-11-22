# Quickstart

Run the Sudoku-hard evaluation. It downloads 0.163 GB. On the test machine the run took 125 seconds.

This procedure was run on 2026-11-21 on Windows 11 with an NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MiB), Python 3.13.5 and `torch 2.11.0+cu128`.

Before you start, complete the [Prerequisites](prerequisites.md).

## Steps

1. Go to the repository root.

   ```bash
   cd ncm
   ```

2. Download the Sudoku-hard checkpoints.

   ```bash
   hf download hazyresearch/ncm-checkpoints --include "sudoku/hard/*" --local-dir artifacts
   ```

   The command writes `artifacts/sudoku/hard/config.yaml`, `generator.pt` (102,560,203 bytes) and `tokenizer.pt` (59,816,209 bytes).

3. Download the Sudoku-hard test boards.

   ```bash
   hf download hazyresearch/ncm-datasets --repo-type dataset --include "sudoku/hard/*" --local-dir data
   ```

   The command writes `data/sudoku/hard/puzzles.pickle` and `data/sudoku/hard/solutions.pickle`.

4. Run the evaluation.

   > **Warning:** Use the folder `artifacts/sudoku/hard`, not `artifacts/sudoku-hard`. The README path `artifacts/<task>` does not match the Hugging Face folder layout.

   ```bash
   python evaluate.py --config config/eval/sudoku-hard.yaml --checkpoint-dir artifacts/sudoku/hard --data-dir data/sudoku/hard
   ```

   Output of the run on 2026-11-21:

   ```text
   [evaluate] 1 shard(s) on GPU(s) 0 -> eval/sudoku-hard/

   merged 1 shards -> 2000 sudoku-hard outputs

    solved %  solved  cell %  valid %
       86.25 1725/2000   98.35    86.25
   ```

   The test run added `--out` with an absolute path in a temporary folder. The first line above shows the default path `eval/sudoku-hard/` in place of that path. All other lines are as printed.

   The paper reports 86.1% for Sudoku hard. This run gave 86.25%.

5. Read the shard log if a step fails.

   ```bash
   cat eval/sudoku-hard/shard_0.log
   ```

## What happened

`evaluate.py` started one `scripts/eval_shard.py` process on GPU 0. The shard loaded the tokenizer (14.90M parameters) and the generator (25.46M parameters). It read 2,000 puzzles of 91 tokens. It ran `net.generate()` in batches of 128, with 12 passes per batch. It decoded the codes to 89 tokens per board and wrote `eval/sudoku-hard/shard_0.pkl`. Then `scripts/eval_merge.py` compared each board with its solution.

The four numbers are: exact-match rate, exact-match count, mean per-cell agreement, and the rate of boards that obey all 27 Sudoku rules (`scripts/eval_merge.py:58-76`).

## Next steps

- [Run an evaluation](../guides/run-an-evaluation.md) for the other 9 configs and all flags.
- [Onboarding](onboarding.md) to learn what the 12 passes do.
- [Projects](../projects/index.md) for experiments with this model.
